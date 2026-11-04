"""Two-ended nonlinear SONIC path realization for TS-seed construction."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from matrix_chem import MolecularGeometry
from matrix_chem.geometry_alignment import kabsch_align
from matrix_chem.symmetry import MolecularSymmetry, symmetrize_molecular_geometry
from matrix_smith.evaluation import build_gic_b_matrix, evaluate_gic_values
from matrix_smith.models import FrozenGIC, GICDefinition, GICPrimitive

from .hybrid_backtransform import hybrid_internal_coordinate_step
from .rigid_pose import RelativeFragmentPose, RigidComplexModel, RigidFragmentPose


SONIC_ENDPOINT_PATH_SCHEMA = "matrix.link.sonic_endpoint_path.v1"
RIGID_POSE_ENDPOINT_PATH_SCHEMA = "matrix.link.rigid_pose_endpoint_path.v1"
MIXED_SONIC_POSE_ENDPOINT_PATH_SCHEMA = "matrix.link.mixed_sonic_pose_endpoint_path.v1"


@dataclass(frozen=True)
class SonicEndpointPathImage:
    index: int
    fraction: float
    coordinates_angstrom: tuple[tuple[float, float, float], ...]
    residual_norm: float
    realization_iterations: int
    branch_rms_angstrom: float


@dataclass(frozen=True)
class SonicEndpointPath:
    schema: str
    images: tuple[SonicEndpointPathImage, ...]
    maximum_residual_norm: float
    maximum_branch_rms_angstrom: float
    symmetry_preserved: bool

    @property
    def central_image(self) -> SonicEndpointPathImage:
        return self.images[len(self.images) // 2]


@dataclass(frozen=True)
class PathEnergyMaximumEstimate:
    """Concave local interpolation around an interior discrete maximum."""

    discrete_index: int
    bracket_indices: tuple[int, int, int]
    fraction: float
    estimated_energy_hartree: float
    curvature_hartree: float


@dataclass(frozen=True)
class EndpointPathCoordinateFrame:
    """Reaction direction and its Euclidean SONIC orthogonal complement."""

    displacement: tuple[float, ...]
    reaction_direction: tuple[float, ...]
    orthogonal_projector: tuple[tuple[float, ...], ...]
    displacement_norm: float
    periodic_coordinate_indices: tuple[int, ...]


@dataclass(frozen=True)
class RigidPoseEndpointPathImage:
    index: int
    fraction: float
    coordinates_angstrom: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class RigidPoseEndpointPath:
    """Shortest-arc SE(3) interpolation between rigid noncovalent minima."""

    schema: str
    images: tuple[RigidPoseEndpointPathImage, ...]
    maximum_rotation_arc_radian: float
    start_reproduction_rms_angstrom: float
    end_reproduction_rms_angstrom: float
    symmetry_preserved: bool = False
    maximum_internal_coordinate_residual: float = 0.0
    maximum_internal_branch_rms_angstrom: float = 0.0
    mixed_intrafragment_sonic: bool = False

    @property
    def central_image(self) -> RigidPoseEndpointPathImage:
        return self.images[len(self.images) // 2]


def realize_sonic_endpoint_path(
    definition: GICDefinition,
    atom_symbols: Sequence[str],
    start_coordinates_angstrom: np.ndarray,
    end_coordinates_angstrom: np.ndarray,
    *,
    images: int = 7,
    symmetry: MolecularSymmetry | None = None,
    residual_tolerance: float = 1.0e-8,
    branch_tolerance_angstrom: float = 1.0e-6,
    max_continuation_increment: float = 0.08,
    max_substeps: int = 64,
    branch_alignment_groups: Sequence[Sequence[int]] | None = None,
) -> SonicEndpointPath:
    """Interpolate SONIC values and realize every target from both endpoints.

    Forward and backward continuations must reach the same Cartesian branch.
    This rejects a formally interpolated coordinate path that crosses a chart
    discontinuity or changes conformation.  When a nontrivial parent symmetry
    is supplied, every corrector step is projected onto that exact symmetry.
    """

    if not isinstance(images, int) or images < 3 or images % 2 == 0:
        raise ValueError("SONIC endpoint path requires an odd number of images >= 3")
    symbols = tuple(str(symbol) for symbol in atom_symbols)
    start = np.asarray(start_coordinates_angstrom, dtype=float)
    end = np.asarray(end_coordinates_angstrom, dtype=float)
    if start.shape != end.shape or start.shape != (len(symbols), 3):
        raise ValueError("SONIC endpoint geometries or atom symbols are incompatible")
    if not np.all(np.isfinite(start)) or not np.all(np.isfinite(end)):
        raise ValueError("SONIC endpoint geometries must be finite")

    values, evaluate, project = _sonic_path_callbacks(definition, symbols, symmetry)

    fractions = tuple(float(value) for value in np.linspace(0.0, 1.0, images))
    start_values = values(start)
    end_values = values(end)
    displacement = sonic_endpoint_displacement(definition, start_values, end_values)
    targets = tuple(start_values + fraction * displacement for fraction in fractions)
    forward = _realize_direction(
        definition,
        start,
        range(1, images - 1),
        targets,
        evaluate,
        values,
        project,
        residual_tolerance=residual_tolerance,
        max_continuation_increment=max_continuation_increment,
        max_substeps=max_substeps,
    )
    backward = _realize_direction(
        definition,
        end,
        range(images - 2, 0, -1),
        targets,
        evaluate,
        values,
        project,
        residual_tolerance=residual_tolerance,
        max_continuation_increment=max_continuation_increment,
        max_substeps=max_substeps,
    )

    output = [
        SonicEndpointPathImage(
            index=0,
            fraction=0.0,
            coordinates_angstrom=_frozen_coordinates(start),
            residual_norm=0.0,
            realization_iterations=0,
            branch_rms_angstrom=0.0,
        )
    ]
    for index in range(1, images - 1):
        forward_coordinates, forward_residual, forward_iterations = forward[index]
        backward_coordinates, backward_residual, backward_iterations = backward[index]
        branch_rms = _branch_rms(
            forward_coordinates,
            backward_coordinates,
            branch_alignment_groups,
        )
        if branch_rms > float(branch_tolerance_angstrom):
            raise RuntimeError(
                "forward/backward SONIC continuations reached different Cartesian branches: "
                f"image={index} rms={branch_rms:.6g} angstrom"
            )
        if forward_residual <= backward_residual:
            selected = forward_coordinates
            residual = forward_residual
            iterations = forward_iterations
        else:
            selected = backward_coordinates
            residual = backward_residual
            iterations = backward_iterations
        output.append(
            SonicEndpointPathImage(
                index=index,
                fraction=fractions[index],
                coordinates_angstrom=_frozen_coordinates(selected),
                residual_norm=residual,
                realization_iterations=iterations,
                branch_rms_angstrom=branch_rms,
            )
        )
    output.append(
        SonicEndpointPathImage(
            index=images - 1,
            fraction=1.0,
            coordinates_angstrom=_frozen_coordinates(end),
            residual_norm=0.0,
            realization_iterations=0,
            branch_rms_angstrom=0.0,
        )
    )
    return SonicEndpointPath(
        schema=SONIC_ENDPOINT_PATH_SCHEMA,
        images=tuple(output),
        maximum_residual_norm=max(image.residual_norm for image in output),
        maximum_branch_rms_angstrom=max(image.branch_rms_angstrom for image in output),
        symmetry_preserved=symmetry is not None,
    )


def fit_sonic_path_energy_maximum(
    fractions: Sequence[float],
    energies_hartree: Sequence[float],
) -> PathEnergyMaximumEstimate:
    """Fit a local parabola through the highest image and its two neighbors."""

    grid = np.asarray(fractions, dtype=float).reshape(-1)
    energies = np.asarray(energies_hartree, dtype=float).reshape(-1)
    if grid.size < 3 or energies.shape != grid.shape:
        raise ValueError("path-energy interpolation requires matching arrays of length >= 3")
    if not np.all(np.isfinite(grid)) or not np.all(np.isfinite(energies)):
        raise ValueError("path-energy interpolation data must be finite")
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError("path fractions must be strictly increasing")
    maximum = float(np.max(energies))
    candidates = np.flatnonzero(energies == maximum)
    if candidates.size != 1:
        raise ValueError("path-energy maximum is not unique")
    index = int(candidates[0])
    if index == 0 or index == grid.size - 1:
        raise ValueError("path-energy maximum is not bracketed by two images")
    if not (energies[index] > energies[index - 1] and energies[index] > energies[index + 1]):
        raise ValueError("highest path image is not a strict local maximum")
    center_fraction = float(grid[index])
    center_energy = float(energies[index])
    local_x = grid[index - 1 : index + 2] - center_fraction
    local_y = energies[index - 1 : index + 2] - center_energy
    quadratic, linear, constant = np.polyfit(local_x, local_y, 2)
    if not np.isfinite(quadratic) or quadratic >= 0.0:
        raise ValueError("local path-energy interpolation is not concave")
    offset = -float(linear) / (2.0 * float(quadratic))
    fraction = center_fraction + offset
    lower = float(grid[index - 1])
    upper = float(grid[index + 1])
    if not lower < fraction < upper:
        raise ValueError("interpolated path-energy maximum lies outside its bracket")
    estimate = center_energy + float(quadratic * offset * offset + linear * offset + constant)
    return PathEnergyMaximumEstimate(
        discrete_index=index,
        bracket_indices=(index - 1, index, index + 1),
        fraction=float(fraction),
        estimated_energy_hartree=estimate,
        curvature_hartree=2.0 * float(quadratic),
    )


def realize_sonic_endpoint_fraction(
    definition: GICDefinition,
    atom_symbols: Sequence[str],
    start_coordinates_angstrom: np.ndarray,
    end_coordinates_angstrom: np.ndarray,
    fraction: float,
    *,
    symmetry: MolecularSymmetry | None = None,
    residual_tolerance: float = 1.0e-8,
    branch_tolerance_angstrom: float = 1.0e-6,
    max_continuation_increment: float = 0.08,
    max_substeps: int = 64,
    branch_alignment_groups: Sequence[Sequence[int]] | None = None,
) -> SonicEndpointPathImage:
    """Realize one arbitrary interpolated SONIC target from both endpoints."""

    value = float(fraction)
    if not np.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError("interpolated SONIC fraction must lie strictly between zero and one")
    symbols = tuple(str(symbol) for symbol in atom_symbols)
    start = np.asarray(start_coordinates_angstrom, dtype=float)
    end = np.asarray(end_coordinates_angstrom, dtype=float)
    if start.shape != end.shape or start.shape != (len(symbols), 3):
        raise ValueError("SONIC endpoint geometries or atom symbols are incompatible")
    values, evaluate, project = _sonic_path_callbacks(definition, symbols, symmetry)
    start_values = values(start)
    end_values = values(end)
    target = start_values + value * sonic_endpoint_displacement(
        definition, start_values, end_values
    )
    targets = (target,)
    forward = _realize_direction(
        definition,
        start,
        (0,),
        targets,
        evaluate,
        values,
        project,
        residual_tolerance=residual_tolerance,
        max_continuation_increment=max_continuation_increment,
        max_substeps=max_substeps,
    )[0]
    backward = _realize_direction(
        definition,
        end,
        (0,),
        targets,
        evaluate,
        values,
        project,
        residual_tolerance=residual_tolerance,
        max_continuation_increment=max_continuation_increment,
        max_substeps=max_substeps,
    )[0]
    forward_coordinates, forward_residual, forward_iterations = forward
    backward_coordinates, backward_residual, backward_iterations = backward
    branch_rms = _branch_rms(
        forward_coordinates,
        backward_coordinates,
        branch_alignment_groups,
    )
    if branch_rms > branch_tolerance_angstrom:
        raise RuntimeError(
            "interpolated SONIC target reaches different Cartesian branches: "
            f"rms={branch_rms:.6g} angstrom"
        )
    if forward_residual <= backward_residual:
        selected, residual, iterations = (
            forward_coordinates,
            forward_residual,
            forward_iterations,
        )
    else:
        selected, residual, iterations = (
            backward_coordinates,
            backward_residual,
            backward_iterations,
        )
    return SonicEndpointPathImage(
        index=-1,
        fraction=value,
        coordinates_angstrom=_frozen_coordinates(selected),
        residual_norm=residual,
        realization_iterations=iterations,
        branch_rms_angstrom=branch_rms,
    )


def realize_sonic_endpoint_progress_from_seed(
    definition: GICDefinition,
    atom_symbols: Sequence[str],
    start_coordinates_angstrom: np.ndarray,
    end_coordinates_angstrom: np.ndarray,
    seed_coordinates_angstrom: np.ndarray,
    fraction: float,
    *,
    residual_tolerance: float = 1.0e-8,
    max_continuation_increment: float = 0.08,
    max_substeps: int = 64,
) -> SonicEndpointPathImage:
    """Advance progress while preserving a relaxed seed's orthogonal displacement."""

    value = float(fraction)
    if not np.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError("continued SONIC fraction must lie strictly between zero and one")
    symbols = tuple(str(symbol) for symbol in atom_symbols)
    start = np.asarray(start_coordinates_angstrom, dtype=float)
    end = np.asarray(end_coordinates_angstrom, dtype=float)
    seed = np.asarray(seed_coordinates_angstrom, dtype=float)
    if start.shape != end.shape or seed.shape != start.shape or start.shape != (
        len(symbols),
        3,
    ):
        raise ValueError("continued SONIC endpoint geometries are incompatible")
    values, evaluate, project = _sonic_path_callbacks(definition, symbols, None)
    start_values = values(start)
    end_values = values(end)
    frame = endpoint_path_coordinate_frame(definition, start_values, end_values)
    target = project_to_endpoint_progress(
        frame,
        start_values,
        values(seed),
        value,
    )
    coordinates, residual, iterations = _realize_direction(
        definition,
        seed,
        (0,),
        (target,),
        evaluate,
        values,
        project,
        residual_tolerance=residual_tolerance,
        max_continuation_increment=max_continuation_increment,
        max_substeps=max_substeps,
    )[0]
    actual_progress = endpoint_path_progress(frame, start_values, values(coordinates))
    # Progress is a normalized projection of every realized SONIC row.  Its
    # roundoff accumulates the endpoint-frame and nonlinear-realization
    # residuals, so comparing it to a bare 1e-8 coordinate residual produced
    # false failures at 1.07e-8 in otherwise converged periodic paths.
    progress_tolerance = max(float(residual_tolerance), 5.0e-8)
    if abs(actual_progress - value) > progress_tolerance:
        raise RuntimeError(
            "continued SONIC realization did not preserve endpoint progress: "
            f"target={value:.12g} actual={actual_progress:.12g} "
            f"tolerance={progress_tolerance:.3g}"
        )
    return SonicEndpointPathImage(
        index=-1,
        fraction=value,
        coordinates_angstrom=_frozen_coordinates(coordinates),
        residual_norm=residual,
        realization_iterations=iterations,
        branch_rms_angstrom=0.0,
    )


def realize_sonic_endpoint_scalar_progress_from_seed(
    definition: GICDefinition,
    atom_symbols: Sequence[str],
    start_coordinates_angstrom: np.ndarray,
    end_coordinates_angstrom: np.ndarray,
    seed_coordinates_angstrom: np.ndarray,
    fraction: float,
    *,
    residual_tolerance: float = 1.0e-8,
    maximum_cartesian_step_angstrom: float = 0.15,
    maximum_iterations: int = 64,
) -> SonicEndpointPathImage:
    """Restore only scalar endpoint progress from a Cartesian path seed.

    A reaction-kernel scaffold need not lie on the generally curved manifold
    obtained by interpolating every internal coordinate.  Its defining
    constraint is instead the single endpoint-progress coordinate.  This
    corrector applies minimum-norm Cartesian Newton steps to that scalar while
    leaving the orthogonal SONIC coordinates free for subsequent relaxation.
    """

    value = float(fraction)
    if not np.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError("scalar SONIC progress fraction must lie strictly between zero and one")
    symbols = tuple(str(symbol) for symbol in atom_symbols)
    start = np.asarray(start_coordinates_angstrom, dtype=float)
    end = np.asarray(end_coordinates_angstrom, dtype=float)
    coordinates = np.asarray(seed_coordinates_angstrom, dtype=float).copy()
    if start.shape != end.shape or coordinates.shape != start.shape or start.shape != (
        len(symbols),
        3,
    ):
        raise ValueError("scalar SONIC progress geometries are incompatible")
    tolerance = float(residual_tolerance)
    maximum_step = float(maximum_cartesian_step_angstrom)
    iterations = int(maximum_iterations)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("scalar SONIC progress tolerance must be finite and positive")
    if not np.isfinite(maximum_step) or maximum_step <= 0.0 or iterations <= 0:
        raise ValueError("scalar SONIC progress corrector bounds must be positive")

    start_values = evaluate_gic_values(definition, coordinates_angstrom=start)
    end_values = evaluate_gic_values(definition, coordinates_angstrom=end)
    frame = endpoint_path_coordinate_frame(definition, start_values, end_values)
    direction = np.asarray(frame.reaction_direction, dtype=float)
    target = value * float(frame.displacement_norm)
    residual = np.inf
    for iteration in range(iterations + 1):
        current_values = evaluate_gic_values(
            definition,
            coordinates_angstrom=coordinates,
        )
        displacement = current_values - start_values
        for index in frame.periodic_coordinate_indices:
            displacement[index] = (displacement[index] + np.pi) % (2.0 * np.pi) - np.pi
        residual = float(target - direction @ displacement)
        if abs(residual) <= tolerance:
            return SonicEndpointPathImage(
                index=-1,
                fraction=value,
                coordinates_angstrom=_frozen_coordinates(coordinates),
                residual_norm=abs(residual),
                realization_iterations=iteration,
                branch_rms_angstrom=0.0,
            )
        b_matrix = np.asarray(
            build_gic_b_matrix(
                definition,
                coordinates_angstrom=coordinates,
            ).rows,
            dtype=float,
        )
        gradient = direction @ b_matrix
        denominator = float(gradient @ gradient)
        if not np.isfinite(denominator) or denominator <= 1.0e-16:
            break
        step = residual * gradient / denominator
        step_norm = float(np.linalg.norm(step))
        if step_norm > maximum_step:
            step *= maximum_step / step_norm
        coordinates += step.reshape(coordinates.shape)
    raise RuntimeError(
        "scalar SONIC endpoint-progress correction did not converge: "
        f"target={value:.12g} residual={abs(residual):.6g}"
    )


def endpoint_path_block_constraint_matrix(
    definition: GICDefinition,
    start_values: Sequence[float] | np.ndarray,
    end_values: Sequence[float] | np.ndarray,
    common_coordinate_count: int,
    *,
    tolerance: float = 1.0e-12,
) -> np.ndarray:
    """Constrain common coordinates individually and reactive progress once."""

    displacement = sonic_endpoint_displacement(definition, start_values, end_values)
    count = int(common_coordinate_count)
    if count < 0 or count >= displacement.size:
        raise ValueError(
            "blockwise endpoint progress needs fewer common coordinates than total rank"
        )
    reactive = np.asarray(displacement[count:], dtype=float)
    norm = float(np.linalg.norm(reactive))
    threshold = float(tolerance)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("blockwise endpoint tolerance must be finite and positive")
    if not np.isfinite(norm) or norm <= threshold:
        raise ValueError("reactive endpoint displacement has zero rank")
    constraints = np.zeros((count + 1, displacement.size), dtype=float)
    if count:
        constraints[:count, :count] = np.eye(count)
    constraints[count, count:] = reactive / norm
    return constraints


def realize_sonic_endpoint_block_progress_from_seed(
    definition: GICDefinition,
    atom_symbols: Sequence[str],
    start_coordinates_angstrom: np.ndarray,
    end_coordinates_angstrom: np.ndarray,
    seed_coordinates_angstrom: np.ndarray,
    fraction: float,
    *,
    common_coordinate_count: int,
    residual_tolerance: float = 1.0e-8,
    maximum_cartesian_step_angstrom: float = 0.15,
    maximum_iterations: int = 64,
) -> SonicEndpointPathImage:
    """Restore affine common coordinates and scalar reactive progress.

    The common SONIC block is followed coordinate by coordinate.  Only the
    endpoint direction inside the remaining reactive block is constrained,
    leaving its orthogonal complement available for transverse relaxation.
    """

    value = float(fraction)
    if not np.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError("blockwise SONIC progress fraction must lie strictly between zero and one")
    symbols = tuple(str(symbol) for symbol in atom_symbols)
    start = np.asarray(start_coordinates_angstrom, dtype=float)
    end = np.asarray(end_coordinates_angstrom, dtype=float)
    coordinates = np.asarray(seed_coordinates_angstrom, dtype=float).copy()
    if start.shape != end.shape or coordinates.shape != start.shape or start.shape != (
        len(symbols),
        3,
    ):
        raise ValueError("blockwise SONIC progress geometries are incompatible")
    tolerance = float(residual_tolerance)
    maximum_step = float(maximum_cartesian_step_angstrom)
    iterations = int(maximum_iterations)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("blockwise SONIC progress tolerance must be finite and positive")
    if not np.isfinite(maximum_step) or maximum_step <= 0.0 or iterations <= 0:
        raise ValueError("blockwise SONIC progress corrector bounds must be positive")

    start_values = evaluate_gic_values(definition, coordinates_angstrom=start)
    end_values = evaluate_gic_values(definition, coordinates_angstrom=end)
    displacement = sonic_endpoint_displacement(definition, start_values, end_values)
    constraints = endpoint_path_block_constraint_matrix(
        definition,
        start_values,
        end_values,
        common_coordinate_count,
    )
    target = value * (constraints @ displacement)
    periodic = set(_periodic_coordinate_indices(definition))
    residual = np.full(constraints.shape[0], np.inf, dtype=float)
    for iteration in range(iterations + 1):
        current_values = evaluate_gic_values(
            definition,
            coordinates_angstrom=coordinates,
        )
        current_displacement = current_values - start_values
        for index in periodic:
            current_displacement[index] = (
                current_displacement[index] + np.pi
            ) % (2.0 * np.pi) - np.pi
        residual = target - constraints @ current_displacement
        residual_norm = float(np.linalg.norm(residual))
        if residual_norm <= tolerance:
            return SonicEndpointPathImage(
                index=-1,
                fraction=value,
                coordinates_angstrom=_frozen_coordinates(coordinates),
                residual_norm=residual_norm,
                realization_iterations=iteration,
                branch_rms_angstrom=0.0,
            )
        b_matrix = np.asarray(
            build_gic_b_matrix(
                definition,
                coordinates_angstrom=coordinates,
            ).rows,
            dtype=float,
        )
        jacobian = constraints @ b_matrix
        if np.linalg.matrix_rank(jacobian, tol=1.0e-12) != constraints.shape[0]:
            break
        step = np.linalg.pinv(jacobian, rcond=1.0e-12) @ residual
        step_norm = float(np.linalg.norm(step))
        if not np.isfinite(step_norm):
            break
        if step_norm > maximum_step:
            step *= maximum_step / step_norm
        coordinates += step.reshape(coordinates.shape)
    raise RuntimeError(
        "blockwise SONIC endpoint-progress correction did not converge: "
        f"target={value:.12g} residual={float(np.linalg.norm(residual)):.6g}"
    )


def sonic_endpoint_displacement(
    definition: GICDefinition,
    start_values: Sequence[float] | np.ndarray,
    end_values: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Return the endpoint SONIC displacement on the shortest periodic branches."""

    start = np.asarray(start_values, dtype=float).reshape(-1)
    end = np.asarray(end_values, dtype=float).reshape(-1)
    expected = len(definition.gics)
    if start.shape != (expected,) or end.shape != (expected,):
        raise ValueError("endpoint SONIC values do not match the frozen definition")
    if not np.all(np.isfinite(start)) or not np.all(np.isfinite(end)):
        raise ValueError("endpoint SONIC values must be finite")
    displacement = end - start
    for index in _periodic_coordinate_indices(definition):
        displacement[index] = (displacement[index] + np.pi) % (2.0 * np.pi) - np.pi
    return displacement


def endpoint_path_coordinate_frame(
    definition: GICDefinition,
    start_values: Sequence[float] | np.ndarray,
    end_values: Sequence[float] | np.ndarray,
    *,
    tolerance: float = 1.0e-12,
) -> EndpointPathCoordinateFrame:
    """Build the fixed reaction direction used by an orthogonally relaxed scan.

    At a prescribed progress value ``s``, a relaxed scan minimizes only in the
    returned projector subspace while keeping the component parallel to the
    endpoint displacement fixed.  The actual energy/gradient optimizer is a
    separate multi-backend layer.
    """

    displacement = sonic_endpoint_displacement(definition, start_values, end_values)
    norm = float(np.linalg.norm(displacement))
    threshold = float(tolerance)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("endpoint-path frame tolerance must be finite and positive")
    if not np.isfinite(norm) or norm <= threshold:
        raise ValueError("endpoint SONIC displacement has zero rank")
    direction = displacement / norm
    projector = np.eye(displacement.size, dtype=float) - np.outer(direction, direction)
    return EndpointPathCoordinateFrame(
        displacement=tuple(float(value) for value in displacement),
        reaction_direction=tuple(float(value) for value in direction),
        orthogonal_projector=tuple(
            tuple(float(value) for value in row) for row in projector
        ),
        displacement_norm=norm,
        periodic_coordinate_indices=_periodic_coordinate_indices(definition),
    )


def orthogonal_endpoint_gradient(
    frame: EndpointPathCoordinateFrame,
    gradient: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Project an energy gradient onto coordinates orthogonal to path progress."""

    vector = np.asarray(gradient, dtype=float).reshape(-1)
    projector = np.asarray(frame.orthogonal_projector, dtype=float)
    if projector.shape != (vector.size, vector.size):
        raise ValueError("endpoint-path gradient and projector dimensions differ")
    if not np.all(np.isfinite(vector)):
        raise ValueError("endpoint-path gradient must be finite")
    return projector @ vector


def endpoint_path_orthogonal_basis(
    frame: EndpointPathCoordinateFrame,
    *,
    tolerance: float = 1.0e-12,
) -> np.ndarray:
    """Return a deterministic orthonormal basis for fixed-progress relaxation."""

    direction = np.asarray(frame.reaction_direction, dtype=float).reshape(-1)
    threshold = float(tolerance)
    if direction.size < 2 or not np.all(np.isfinite(direction)):
        raise ValueError("endpoint path needs at least two finite coordinates")
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("orthogonal-basis tolerance must be finite and positive")
    norm = float(np.linalg.norm(direction))
    if abs(norm - 1.0) > threshold:
        raise ValueError("endpoint reaction direction must be normalized")
    basis = endpoint_path_constraint_orthogonal_basis(
        direction.reshape(1, -1),
        tolerance=threshold,
    )
    return basis


def endpoint_path_constraint_orthogonal_basis(
    constraints: Sequence[Sequence[float]] | np.ndarray,
    *,
    tolerance: float = 1.0e-12,
) -> np.ndarray:
    """Return a deterministic basis for the null space of path constraints."""

    matrix = np.asarray(constraints, dtype=float)
    threshold = float(tolerance)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 2:
        raise ValueError("endpoint path constraints must be a non-empty two-dimensional matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("endpoint path constraints must be finite")
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("constraint-basis tolerance must be finite and positive")
    _left, singular, right = np.linalg.svd(matrix, full_matrices=True)
    rank = int(np.count_nonzero(singular > threshold))
    if rank != matrix.shape[0] or rank >= matrix.shape[1]:
        raise ValueError("endpoint path constraints must be full row rank and leave a complement")
    basis = np.asarray(right[rank:, :].T, dtype=float)
    for column in range(basis.shape[1]):
        pivot = int(np.argmax(np.abs(basis[:, column])))
        if basis[pivot, column] < 0.0:
            basis[:, column] *= -1.0
    if (
        np.linalg.matrix_rank(basis, tol=threshold) != matrix.shape[1] - rank
        or float(np.linalg.norm(matrix @ basis)) > 10.0 * threshold
        or float(np.linalg.norm(basis.T @ basis - np.eye(matrix.shape[1] - rank)))
        > 10.0 * threshold
    ):
        raise RuntimeError("failed to construct the endpoint constraint complement")
    return basis


def project_to_endpoint_progress(
    frame: EndpointPathCoordinateFrame,
    start_values: Sequence[float] | np.ndarray,
    current_values: Sequence[float] | np.ndarray,
    fraction: float,
) -> np.ndarray:
    """Restore the affine progress constraint while retaining orthogonal motion."""

    start = np.asarray(start_values, dtype=float).reshape(-1)
    current = np.asarray(current_values, dtype=float).reshape(-1)
    direction = np.asarray(frame.reaction_direction, dtype=float)
    if start.shape != direction.shape or current.shape != direction.shape:
        raise ValueError("endpoint-path progress values have incompatible dimensions")
    value = float(fraction)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("endpoint-path progress fraction must lie between zero and one")
    target_projection = value * float(frame.displacement_norm)
    current_projection = float(direction @ (current - start))
    return current + (target_projection - current_projection) * direction


def endpoint_path_progress(
    frame: EndpointPathCoordinateFrame,
    start_values: Sequence[float] | np.ndarray,
    current_values: Sequence[float] | np.ndarray,
) -> float:
    """Measure normalized progress along the frozen endpoint direction."""

    start = np.asarray(start_values, dtype=float).reshape(-1)
    current = np.asarray(current_values, dtype=float).reshape(-1)
    direction = np.asarray(frame.reaction_direction, dtype=float)
    if start.shape != direction.shape or current.shape != direction.shape:
        raise ValueError("endpoint-path progress values have incompatible dimensions")
    return float(direction @ (current - start) / float(frame.displacement_norm))


def realize_rigid_pose_endpoint_path(
    definition: GICDefinition,
    start_coordinates_angstrom: np.ndarray,
    end_coordinates_angstrom: np.ndarray,
    *,
    images: int = 7,
    rigid_reproduction_tolerance_angstrom: float = 1.0e-4,
) -> RigidPoseEndpointPath:
    """Interpolate noncovalent fragment poses by translation and quaternion SLERP.

    This fast path is deliberately fail-closed: both endpoint minima must be
    representable by the same frozen rigid-fragment model.  Flexible endpoint
    changes belong to the mixed intrafragment-SONIC/SE(3) path, not to an
    accidental Cartesian distortion of either partner.
    """

    if not isinstance(images, int) or images < 3 or images % 2 == 0:
        raise ValueError("rigid-pose endpoint path requires an odd number of images >= 3")
    start = np.asarray(start_coordinates_angstrom, dtype=float)
    end = np.asarray(end_coordinates_angstrom, dtype=float)
    if start.shape != end.shape or start.ndim != 2 or start.shape[1] != 3:
        raise ValueError("rigid-pose endpoint geometries must have matching natom x 3 shapes")
    if not np.all(np.isfinite(start)) or not np.all(np.isfinite(end)):
        raise ValueError("rigid-pose endpoint geometries must be finite")
    tolerance = float(rigid_reproduction_tolerance_angstrom)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("rigid-pose reproduction tolerance must be finite and positive")

    model = _rigid_geometry_model(definition)
    start_poses = model.extract_poses(start)
    end_poses = model.extract_poses(end)
    realized_start = model.realize(start_poses)
    realized_end = model.realize(end_poses)
    start_rms = _aligned_rms(realized_start, start)
    end_rms = _aligned_rms(realized_end, end)
    if max(start_rms, end_rms) > tolerance:
        raise ValueError(
            "noncovalent endpoints are not rigidly representable by one frozen pose model: "
            f"start_rms={start_rms:.6g} end_rms={end_rms:.6g} angstrom"
        )

    arcs = tuple(
        _quaternion_arc(left.quaternion_wxyz, right.quaternion_wxyz)
        for left, right in zip(start_poses, end_poses, strict=True)
    )
    output: list[RigidPoseEndpointPathImage] = []
    for index, fraction in enumerate(np.linspace(0.0, 1.0, images)):
        poses = _interpolate_rigid_poses(start_poses, end_poses, float(fraction))
        output.append(
            RigidPoseEndpointPathImage(
                index=index,
                fraction=float(fraction),
                coordinates_angstrom=_frozen_coordinates(model.realize(poses)),
            )
        )
    return RigidPoseEndpointPath(
        schema=RIGID_POSE_ENDPOINT_PATH_SCHEMA,
        images=tuple(output),
        maximum_rotation_arc_radian=max(arcs, default=0.0),
        start_reproduction_rms_angstrom=start_rms,
        end_reproduction_rms_angstrom=end_rms,
    )


def realize_rigid_pose_endpoint_fraction(
    definition: GICDefinition,
    start_coordinates_angstrom: np.ndarray,
    end_coordinates_angstrom: np.ndarray,
    fraction: float,
    *,
    rigid_reproduction_tolerance_angstrom: float = 1.0e-4,
) -> RigidPoseEndpointPathImage:
    """Realize one arbitrary point on the shortest-arc rigid-pose path."""

    value = float(fraction)
    if not np.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError("interpolated rigid-pose fraction must lie strictly between zero and one")
    # Apply exactly the same endpoint representability gate as a complete path.
    realize_rigid_pose_endpoint_path(
        definition,
        start_coordinates_angstrom,
        end_coordinates_angstrom,
        images=3,
        rigid_reproduction_tolerance_angstrom=rigid_reproduction_tolerance_angstrom,
    )
    model = _rigid_geometry_model(definition)
    start_poses = model.extract_poses(np.asarray(start_coordinates_angstrom, dtype=float))
    end_poses = model.extract_poses(np.asarray(end_coordinates_angstrom, dtype=float))
    coordinates = model.realize(_interpolate_rigid_poses(start_poses, end_poses, value))
    return RigidPoseEndpointPathImage(
        index=-1,
        fraction=value,
        coordinates_angstrom=_frozen_coordinates(coordinates),
    )


def realize_mixed_sonic_rigid_pose_endpoint_path(
    definition: GICDefinition,
    atom_symbols: Sequence[str],
    start_coordinates_angstrom: np.ndarray,
    end_coordinates_angstrom: np.ndarray,
    *,
    images: int = 7,
) -> RigidPoseEndpointPath:
    """Interpolate flexible fragments in SONIC and their relative pose by SLERP."""

    model = _rigid_geometry_model(definition)
    internal_definition = _intrafragment_sonic_definition(definition, model)
    fragment_groups = (
        model.blocks[0].reference_atom_indices,
        *(block.atom_indices for block in model.blocks),
    )
    sonic_path = realize_sonic_endpoint_path(
        internal_definition,
        atom_symbols,
        start_coordinates_angstrom,
        end_coordinates_angstrom,
        images=images,
        symmetry=None,
        branch_alignment_groups=fragment_groups,
    )
    start_poses = model.extract_relative_poses(
        np.asarray(start_coordinates_angstrom, dtype=float)
    )
    end_poses = model.extract_relative_poses(
        np.asarray(end_coordinates_angstrom, dtype=float)
    )
    arcs = tuple(
        _quaternion_arc(left.quaternion_wxyz, right.quaternion_wxyz)
        for left, right in zip(start_poses, end_poses, strict=True)
    )
    output = tuple(
        RigidPoseEndpointPathImage(
            index=image.index,
            fraction=image.fraction,
            coordinates_angstrom=_frozen_coordinates(
                model.realize_relative_poses_from_base(
                    np.asarray(image.coordinates_angstrom, dtype=float),
                    _interpolate_relative_poses(
                        start_poses,
                        end_poses,
                        image.fraction,
                    ),
                )
            ),
        )
        for image in sonic_path.images
    )
    start_rms = _aligned_rms(
        np.asarray(output[0].coordinates_angstrom, dtype=float),
        np.asarray(start_coordinates_angstrom, dtype=float),
    )
    end_rms = _aligned_rms(
        np.asarray(output[-1].coordinates_angstrom, dtype=float),
        np.asarray(end_coordinates_angstrom, dtype=float),
    )
    return RigidPoseEndpointPath(
        schema=MIXED_SONIC_POSE_ENDPOINT_PATH_SCHEMA,
        images=output,
        maximum_rotation_arc_radian=max(arcs, default=0.0),
        start_reproduction_rms_angstrom=start_rms,
        end_reproduction_rms_angstrom=end_rms,
        maximum_internal_coordinate_residual=sonic_path.maximum_residual_norm,
        maximum_internal_branch_rms_angstrom=(
            sonic_path.maximum_branch_rms_angstrom
        ),
        mixed_intrafragment_sonic=True,
    )


def realize_mixed_sonic_rigid_pose_endpoint_fraction(
    definition: GICDefinition,
    atom_symbols: Sequence[str],
    start_coordinates_angstrom: np.ndarray,
    end_coordinates_angstrom: np.ndarray,
    fraction: float,
) -> RigidPoseEndpointPathImage:
    """Realize one flexible-SONIC/relative-quaternion interpolation point."""

    value = float(fraction)
    model = _rigid_geometry_model(definition)
    internal_definition = _intrafragment_sonic_definition(definition, model)
    fragment_groups = (
        model.blocks[0].reference_atom_indices,
        *(block.atom_indices for block in model.blocks),
    )
    base = realize_sonic_endpoint_fraction(
        internal_definition,
        atom_symbols,
        start_coordinates_angstrom,
        end_coordinates_angstrom,
        value,
        symmetry=None,
        branch_alignment_groups=fragment_groups,
    )
    start_poses = model.extract_relative_poses(
        np.asarray(start_coordinates_angstrom, dtype=float)
    )
    end_poses = model.extract_relative_poses(
        np.asarray(end_coordinates_angstrom, dtype=float)
    )
    coordinates = model.realize_relative_poses_from_base(
        np.asarray(base.coordinates_angstrom, dtype=float),
        _interpolate_relative_poses(start_poses, end_poses, value),
    )
    return RigidPoseEndpointPathImage(
        index=-1,
        fraction=value,
        coordinates_angstrom=_frozen_coordinates(coordinates),
    )


def _intrafragment_sonic_definition(
    definition: GICDefinition,
    model: RigidComplexModel,
) -> GICDefinition:
    """Remove rigid-pose rows from the SONIC block used for shape interpolation.

    Relative translations and rotations are realized separately by linear
    translation and quaternion SLERP.  Leaving those rows in the nonlinear
    SONIC corrector both duplicates the pose interpolation and can make a row
    switch dynamically between solved and protected sets.
    """

    del model
    primitive_by_id = {item.identifier: item for item in definition.primitives}
    pose_functions = {"FC_DIST", "FCA_DIST", "FTRANS", "FLIN_TRANS", "FROT", "FAXIS"}
    internal_gics = tuple(
        gic
        for gic in definition.gics
        if not any(
            primitive_by_id[primitive_id].function in pose_functions
            for primitive_id, _coefficient in gic.coefficients
            if primitive_id in primitive_by_id
        )
    )
    if not internal_gics:
        raise ValueError("mixed SONIC/rigid-pose path has no intrafragment coordinates")
    retained_ids = {gic.identifier for gic in internal_gics}
    periodic = tuple(
        record
        for record in definition.periodic_coordinate_estimates
        if record.coordinate_identifier in retained_ids
    )
    return replace(
        definition,
        gics=internal_gics,
        target_rank=len(internal_gics),
        rank=len(internal_gics),
        periodic_coordinate_estimates=periodic,
        primitive_b_matrix_sha256="",
        wilson_tangent_rank=0,
        wilson_tangent_singular_min=0.0,
        wilson_tangent_singular_max=0.0,
    )


def _rigid_geometry_model(definition: GICDefinition) -> RigidComplexModel:
    """Compile the SE(3) endpoint model from the complete primitive pose pool.

    A nonredundant minimum/exploration chart replaces the radial FTRANS row by
    FC_DIST.  That is the right optimization basis but it intentionally leaves
    no complete six-row FTRANS/FROT subset among the selected GICs.  Endpoint
    quaternion interpolation needs only fragment membership and local frames,
    so construct a private complete pose chart from the retained primitive
    pool.  This does not alter the SONIC definition or reintroduce a redundant
    coordinate into optimization.
    """

    direct = RigidComplexModel.try_from_definition(definition)
    if direct is not None:
        return direct
    grouped: dict[tuple[tuple[int, ...], tuple[int, ...]], list[GICPrimitive]] = {}
    for primitive in definition.primitives:
        if primitive.function in {"FC_DIST", "FTRANS", "FROT"}:
            grouped.setdefault(
                (tuple(primitive.atoms), tuple(primitive.ref_atoms)), []
            ).append(primitive)
    selected: list[GICPrimitive] = []
    for (_atoms, _reference_atoms), pool in sorted(grouped.items()):
        rotations = {item.mode: item for item in pool if item.function == "FROT"}
        translations = {item.mode: item for item in pool if item.function == "FTRANS"}
        if set(rotations) != {0, 1, 2} or not translations:
            continue
        template = translations[min(translations)]
        rotation_template = rotations[0]
        for mode in range(3):
            if mode not in translations:
                translations[mode] = replace(
                    template,
                    identifier=f"__POSE_FTRANS_{len(selected):04d}_{mode}",
                    name=f"PoseT{len(selected):04d}{mode}",
                    mode=mode,
                    ref_frame_atoms=(
                        template.ref_frame_atoms
                        or rotation_template.ref_frame_atoms
                    ),
                    provenance="LINK_PRIVATE_ENDPOINT_POSE_COMPLETION",
                )
        selected.extend(translations[mode] for mode in range(3))
        selected.extend(rotations[mode] for mode in range(3))
    if not selected:
        raise ValueError("definition has no complete fragment pose primitive pool")
    gics = tuple(
        FrozenGIC(
            identifier=f"__POSE_GIC_{index:04d}",
            name=f"Pose{index:04d}",
            family=primitive.family,
            irrep="A",
            primitive_id=primitive.identifier,
            gaussian_expression=primitive.gaussian_expression(),
            coefficients=((primitive.identifier, 1.0),),
        )
        for index, primitive in enumerate(selected)
    )
    private_definition = replace(
        definition,
        primitives=tuple(selected),
        gics=gics,
        rank=len(gics),
        target_rank=len(gics),
    )
    return RigidComplexModel.from_definition(private_definition)


def _branch_rms(
    forward: np.ndarray,
    backward: np.ndarray,
    groups: Sequence[Sequence[int]] | None,
) -> float:
    """Compare continuation branches globally or by independent fragments."""

    left = np.asarray(forward, dtype=float)
    right = np.asarray(backward, dtype=float)
    if groups is None:
        aligned = kabsch_align(right, left)
        return float(np.sqrt(np.mean(np.sum((aligned - left) ** 2, axis=1))))
    squared = 0.0
    count = 0
    covered: set[int] = set()
    for raw_group in groups:
        group = tuple(int(index) for index in raw_group)
        if not group or covered.intersection(group):
            raise ValueError("branch-alignment fragment groups must be nonempty and disjoint")
        covered.update(group)
        indices = np.asarray(group, dtype=int)
        aligned = kabsch_align(right[indices], left[indices])
        squared += float(np.sum((aligned - left[indices]) ** 2))
        count += len(group)
    if covered != set(range(left.shape[0])):
        raise ValueError("branch-alignment fragment groups must cover every atom")
    return float(np.sqrt(squared / count))


def _interpolate_rigid_poses(
    start_poses: Sequence[RigidFragmentPose],
    end_poses: Sequence[RigidFragmentPose],
    fraction: float,
) -> tuple[RigidFragmentPose, ...]:
    return tuple(
        RigidFragmentPose(
            (1.0 - fraction) * left.translation_angstrom
            + fraction * right.translation_angstrom,
            _quaternion_slerp(
                left.quaternion_wxyz,
                right.quaternion_wxyz,
                fraction,
            ),
        )
        for left, right in zip(start_poses, end_poses, strict=True)
    )


def _interpolate_relative_poses(
    start_poses: Sequence[RelativeFragmentPose],
    end_poses: Sequence[RelativeFragmentPose],
    fraction: float,
) -> tuple[RelativeFragmentPose, ...]:
    return tuple(
        RelativeFragmentPose(
            (1.0 - fraction) * left.translation_angstrom
            + fraction * right.translation_angstrom,
            _quaternion_slerp(
                left.quaternion_wxyz,
                right.quaternion_wxyz,
                fraction,
            ),
        )
        for left, right in zip(start_poses, end_poses, strict=True)
    )


def _periodic_coordinate_indices(definition: GICDefinition) -> tuple[int, ...]:
    identifiers = {
        str(record.coordinate_identifier)
        for record in definition.periodic_coordinate_estimates
        if str(record.coordinate_domain).upper() == "PERIODIC_2PI"
    }
    return tuple(
        index for index, gic in enumerate(definition.gics) if gic.identifier in identifiers
    )


def _aligned_rms(moving: np.ndarray, reference: np.ndarray) -> float:
    aligned = kabsch_align(np.asarray(moving, dtype=float), np.asarray(reference, dtype=float))
    return float(np.sqrt(np.mean(np.sum((aligned - reference) ** 2, axis=1))))


def _quaternion_arc(left: np.ndarray, right: np.ndarray) -> float:
    dot = abs(float(np.dot(np.asarray(left, dtype=float), np.asarray(right, dtype=float))))
    return 2.0 * float(np.arccos(np.clip(dot, -1.0, 1.0)))


def _quaternion_slerp(left: np.ndarray, right: np.ndarray, fraction: float) -> np.ndarray:
    start = np.asarray(left, dtype=float).reshape(4)
    end = np.asarray(right, dtype=float).reshape(4)
    dot = float(np.dot(start, end))
    if dot < 0.0:
        end = -end
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 1.0 - 1.0e-10:
        interpolated = (1.0 - fraction) * start + fraction * end
        return interpolated / np.linalg.norm(interpolated)
    angle = float(np.arccos(dot))
    scale = float(np.sin(angle))
    return (
        np.sin((1.0 - fraction) * angle) * start
        + np.sin(fraction * angle) * end
    ) / scale


def _realize_direction(
    definition: GICDefinition,
    initial_coordinates: np.ndarray,
    indices,
    targets: Sequence[np.ndarray],
    evaluate,
    evaluate_values,
    project,
    *,
    residual_tolerance: float,
    max_continuation_increment: float,
    max_substeps: int,
) -> dict[int, tuple[np.ndarray, float, int]]:
    coordinates = np.asarray(initial_coordinates, dtype=float).copy()
    output: dict[int, tuple[np.ndarray, float, int]] = {}
    for index in indices:
        result = hybrid_internal_coordinate_step(
            definition,
            coordinates,
            targets[index],
            evaluate,
            evaluate_values=evaluate_values,
            project_coordinates=project,
            tolerance=residual_tolerance,
            max_continuation_increment=max_continuation_increment,
            max_substeps=max_substeps,
        )
        residual = float(np.linalg.norm(result.residual))
        if not result.converged or residual > residual_tolerance:
            raise RuntimeError(
                "SONIC endpoint-path back-transformation did not converge: "
                f"image={index} residual={residual:.6g}"
            )
        coordinates = np.asarray(result.coordinates_angstrom, dtype=float)
        output[int(index)] = (
            coordinates.copy(),
            residual,
            int(result.corrector_iterations),
        )
    return output


def _sonic_path_callbacks(
    definition: GICDefinition,
    symbols: tuple[str, ...],
    symmetry: MolecularSymmetry | None,
):
    def values(coordinates: np.ndarray) -> np.ndarray:
        return evaluate_gic_values(definition, coordinates_angstrom=coordinates)

    def evaluate(coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return values(coordinates), np.asarray(
            build_gic_b_matrix(definition, coordinates_angstrom=coordinates).rows,
            dtype=float,
        )

    def project(coordinates: np.ndarray) -> np.ndarray:
        if symmetry is None:
            return np.asarray(coordinates, dtype=float)
        geometry = MolecularGeometry(atoms=symbols, coordinates_angstrom=coordinates)
        return symmetrize_molecular_geometry(
            geometry,
            symmetry,
            force_projection=True,
        ).coordinates_angstrom

    return values, evaluate, project


def _frozen_coordinates(coordinates: np.ndarray) -> tuple[tuple[float, float, float], ...]:
    return tuple(tuple(float(value) for value in row) for row in coordinates)


__all__ = [
    "MIXED_SONIC_POSE_ENDPOINT_PATH_SCHEMA",
    "RIGID_POSE_ENDPOINT_PATH_SCHEMA",
    "SONIC_ENDPOINT_PATH_SCHEMA",
    "EndpointPathCoordinateFrame",
    "PathEnergyMaximumEstimate",
    "RigidPoseEndpointPath",
    "RigidPoseEndpointPathImage",
    "SonicEndpointPath",
    "SonicEndpointPathImage",
    "endpoint_path_progress",
    "endpoint_path_coordinate_frame",
    "endpoint_path_block_constraint_matrix",
    "endpoint_path_constraint_orthogonal_basis",
    "endpoint_path_orthogonal_basis",
    "orthogonal_endpoint_gradient",
    "project_to_endpoint_progress",
    "realize_mixed_sonic_rigid_pose_endpoint_fraction",
    "realize_mixed_sonic_rigid_pose_endpoint_path",
    "realize_rigid_pose_endpoint_fraction",
    "realize_rigid_pose_endpoint_path",
    "realize_sonic_endpoint_path",
    "fit_sonic_path_energy_maximum",
    "realize_sonic_endpoint_fraction",
    "realize_sonic_endpoint_block_progress_from_seed",
    "realize_sonic_endpoint_progress_from_seed",
    "realize_sonic_endpoint_scalar_progress_from_seed",
    "sonic_endpoint_displacement",
]
